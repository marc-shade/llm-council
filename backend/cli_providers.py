"""CLI-based LLM providers for local AI tools.

Wraps Claude Code, OpenAI Codex CLI, and Gemini CLI as async providers
that can work together in a council configuration.
"""

import asyncio
import os
import tempfile
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ProviderType(Enum):
    """Supported CLI providers."""
    CLAUDE = "claude"
    CODEX = "codex"
    GEMINI = "gemini"


@dataclass
class CLIProvider:
    """Configuration for a CLI-based LLM provider."""
    name: str
    provider_type: ProviderType
    display_name: str
    command: str
    args_template: List[str]
    timeout: float = 120.0

    def get_command_args(self, prompt: str, prompt_file: Optional[str] = None) -> List[str]:
        """Build command arguments for the provider."""
        args = [self.command]
        for arg in self.args_template:
            if arg == "{prompt}":
                args.append(prompt)
            elif arg == "{prompt_file}":
                args.append(prompt_file or prompt)
            else:
                args.append(arg)
        return args


# Pre-configured providers
PROVIDERS = {
    "claude": CLIProvider(
        name="claude",
        provider_type=ProviderType.CLAUDE,
        display_name="Claude Code (Anthropic)",
        command="claude",
        args_template=["-p", "{prompt}", "--print"],
        timeout=180.0
    ),
    "codex": CLIProvider(
        name="codex",
        provider_type=ProviderType.CODEX,
        display_name="Codex CLI (OpenAI)",
        command="codex",
        args_template=["{prompt}"],
        timeout=180.0
    ),
    "gemini": CLIProvider(
        name="gemini",
        provider_type=ProviderType.GEMINI,
        display_name="Gemini CLI (Google)",
        command="gemini",
        args_template=["-p", "{prompt}"],
        timeout=180.0
    ),
}


async def query_cli_provider(
    provider_name: str,
    prompt: str,
    timeout: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    """
    Query a single CLI provider asynchronously.

    Args:
        provider_name: Name of the provider (claude, codex, gemini)
        prompt: The prompt to send
        timeout: Optional timeout override

    Returns:
        Response dict with 'content' key, or None if failed
    """
    provider = PROVIDERS.get(provider_name)
    if not provider:
        print(f"Unknown provider: {provider_name}")
        return None

    effective_timeout = timeout or provider.timeout

    try:
        # For long prompts, use a temp file
        if len(prompt) > 4000:
            return await _query_with_file(provider, prompt, effective_timeout)
        else:
            return await _query_direct(provider, prompt, effective_timeout)

    except asyncio.TimeoutError:
        print(f"Timeout querying {provider_name} after {effective_timeout}s")
        return None
    except Exception as e:
        print(f"Error querying {provider_name}: {e}")
        return None


async def _query_direct(
    provider: CLIProvider,
    prompt: str,
    timeout: float
) -> Optional[Dict[str, Any]]:
    """Query provider with prompt as argument."""

    # Build command based on provider type
    if provider.provider_type == ProviderType.CLAUDE:
        cmd = ["claude", "-p", prompt, "--print"]
    elif provider.provider_type == ProviderType.CODEX:
        # Codex requires 'exec' subcommand for non-interactive mode
        cmd = ["codex", "exec", prompt]
    elif provider.provider_type == ProviderType.GEMINI:
        # Use positional prompt (--prompt is deprecated)
        cmd = ["gemini", prompt]
    else:
        return None

    # Set up environment
    env = os.environ.copy()
    env["NO_COLOR"] = "1"  # Disable color codes in output

    # For Claude, set ANTHROPIC_API_KEY to empty string to force OAuth/subscription auth
    # (Just removing the env var isn't enough - Claude CLI may read from config files)
    if provider.provider_type == ProviderType.CLAUDE:
        env["ANTHROPIC_API_KEY"] = ""

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env
    )

    stdout, stderr = await asyncio.wait_for(
        process.communicate(),
        timeout=timeout
    )

    if process.returncode != 0:
        error_msg = stderr.decode() if stderr else "Unknown error"
        print(f"Provider {provider.name} returned error: {error_msg}")
        # Still return output if there is any
        if stdout:
            return {"content": stdout.decode().strip()}
        return None

    content = stdout.decode().strip()
    return {"content": content} if content else None


async def _query_with_file(
    provider: CLIProvider,
    prompt: str,
    timeout: float
) -> Optional[Dict[str, Any]]:
    """Query provider using a temp file for long prompts."""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(prompt)
        prompt_file = f.name

    try:
        # Use file-based input
        if provider.provider_type == ProviderType.CLAUDE:
            # Claude can read from stdin
            cmd = ["claude", "--print"]

            # Set up environment - empty API key to force OAuth/subscription auth
            env = os.environ.copy()
            env["NO_COLOR"] = "1"
            env["ANTHROPIC_API_KEY"] = ""

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=prompt.encode()),
                timeout=timeout
            )
        else:
            # For others, pass prompt directly (they handle it)
            return await _query_direct(provider, prompt, timeout)

        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            print(f"Provider {provider.name} returned error: {error_msg}")
            if stdout:
                return {"content": stdout.decode().strip()}
            return None

        content = stdout.decode().strip()
        return {"content": content} if content else None

    finally:
        os.unlink(prompt_file)


async def query_providers_parallel(
    provider_names: List[str],
    prompt: str
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple CLI providers in parallel.

    Args:
        provider_names: List of provider names to query
        prompt: The prompt to send to each

    Returns:
        Dict mapping provider name to response dict (or None if failed)
    """
    tasks = [query_cli_provider(name, prompt) for name in provider_names]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    results = {}
    for name, response in zip(provider_names, responses):
        if isinstance(response, Exception):
            print(f"Exception from {name}: {response}")
            results[name] = None
        else:
            results[name] = response

    return results


# Compatibility layer - matches openrouter.py interface
async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a model (CLI provider) - compatible with openrouter.py interface.

    Args:
        model: Provider name (claude, codex, gemini)
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' key, or None if failed
    """
    # Convert messages to a single prompt
    prompt = _messages_to_prompt(messages)
    return await query_cli_provider(model, prompt, timeout)


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models (CLI providers) in parallel.
    Compatible with openrouter.py interface.

    Args:
        models: List of provider names
        messages: List of message dicts to send to each

    Returns:
        Dict mapping provider name to response dict (or None if failed)
    """
    prompt = _messages_to_prompt(messages)
    return await query_providers_parallel(models, prompt)


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Convert chat messages to a single prompt string."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(content)
    return "\n\n".join(parts)


def get_available_providers() -> List[str]:
    """Return list of available provider names."""
    return list(PROVIDERS.keys())


def get_provider_info(name: str) -> Optional[Dict[str, Any]]:
    """Get information about a provider."""
    provider = PROVIDERS.get(name)
    if not provider:
        return None
    return {
        "name": provider.name,
        "display_name": provider.display_name,
        "type": provider.provider_type.value,
        "timeout": provider.timeout
    }


# Quick test
if __name__ == "__main__":
    async def test():
        print("Testing CLI providers...")
        print(f"Available: {get_available_providers()}")

        # Test each provider
        for name in ["claude", "codex", "gemini"]:
            print(f"\nTesting {name}...")
            result = await query_cli_provider(name, "Say hello in exactly 5 words.")
            if result:
                print(f"  Response: {result['content'][:100]}...")
            else:
                print(f"  Failed to get response")

    asyncio.run(test())
