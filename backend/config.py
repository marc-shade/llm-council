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
# Reordered 2026-06-03: the claude CLI subprocess reliably fails ("claude failed
# to respond") when spawned nested from this backend, so it is removed from the
# active council until that nested-invocation is fixed separately. red_team uses
# models[0]=blue and models[1]=red plus CLI_CHAIRMAN_MODEL, so positions 0/1 and
# the chairman must be VERIFIED-working providers (gemini, ollama). codex re-wired
# to the subprocess form (cli_providers.py) and added back as the OpenAI seat.
CLI_COUNCIL_MODELS = [
    "gemini",  # Gemini CLI (Google) - blue team / chairman (verified working 2026-06-03)
    "ollama",  # Ollama Cloud gpt-oss:120b-cloud (OpenAI-compat) - red team (verified)
    "codex",  # Codex CLI (OpenAI gpt-5.5) - re-wired 2026-06-03 to subprocess form
    "claude",  # Claude Code (Anthropic) - restored 2026-06-03 after binary-path fix (was off PATH under launchd)
    "hermes",  # Hermes Agent (MiniMax-M3 via Hermes config)
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
    "hermes": "Hermes Agent (MiniMax-M3)",
    "llama_server": "llama-server (Local)",
}

# Timeouts for each provider (CLI tools can be slower)
PROVIDER_TIMEOUTS = {
    "claude": 180.0,
    "ollama": 120.0,
    "gemini": 180.0,
    "hermes": 300.0,
    "llama_server": 300.0,
}
