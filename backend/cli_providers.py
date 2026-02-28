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
    OLLAMA = "ollama"


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
    "ollama": CLIProvider(
        name="ollama",
        provider_type=ProviderType.OLLAMA,
        display_name="Ollama (Local)",
        command="ollama",
        args_template=["run", "{model}", "{prompt}"],
        timeout=300.0  # Local models may need more time
    ),
}

# Default models - will be updated dynamically where possible
DEFAULT_MODELS = {
    "claude": "claude-sonnet-4-20250514",  # Safe default
    "codex": "o3",  # Deprecated - using ollama instead
    "gemini": "gemini-2.5-pro",
    "ollama": "gpt-oss:120b-cloud",  # OpenAI-compatible model via Ollama Cloud
}

# Model cache - populated dynamically
_model_cache: Dict[str, List[Dict[str, str]]] = {}
_cache_timestamp: Dict[str, float] = {}
_CACHE_TTL = 300  # 5 minutes


async def _fetch_ollama_models() -> List[Dict[str, str]]:
    """Dynamically fetch installed Ollama models via `ollama list`."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ollama", "list",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)

        models = []
        lines = stdout.decode().strip().split('\n')
        for line in lines[1:]:  # Skip header line
            if line.strip():
                parts = line.split()
                if parts:
                    model_id = parts[0]
                    # Clean up model name for display
                    name = model_id.replace(':', ' ').replace('-', ' ').title()
                    models.append({"id": model_id, "name": name})

        return models if models else [{"id": "llama3.2:latest", "name": "Llama 3.2 (default)"}]
    except Exception as e:
        print(f"Failed to fetch Ollama models: {e}")
        return [{"id": "llama3.2:latest", "name": "Llama 3.2 (default)"}]


async def _fetch_claude_models() -> List[Dict[str, str]]:
    """Fetch Claude models from API."""
    try:
        # Try to get models from Anthropic API
        import httpx
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://api.anthropic.com/v1/models",
                    headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                    timeout=10.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    models = []
                    for m in data.get("data", []):
                        model_id = m.get("id", "")
                        # Filter to chat models only
                        if "claude" in model_id.lower():
                            name = model_id.replace("-", " ").replace("claude", "Claude").title()
                            models.append({"id": model_id, "name": name})
                    if models:
                        return sorted(models, key=lambda x: x["id"], reverse=True)[:6]
    except Exception as e:
        print(f"Failed to fetch Claude models from API: {e}")

    # Fallback - these are the current models as of the API
    return [
        {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4"},
        {"id": "claude-opus-4-20250514", "name": "Claude Opus 4"},
    ]


async def _fetch_openai_models() -> List[Dict[str, str]]:
    """Fetch OpenAI models from API."""
    try:
        import httpx
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=10.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    models = []
                    # Filter to relevant models
                    relevant = ["gpt-4", "gpt-5", "o1", "o3", "o4"]
                    for m in data.get("data", []):
                        model_id = m.get("id", "")
                        if any(r in model_id.lower() for r in relevant):
                            name = model_id.replace("-", " ").upper() if model_id.startswith("o") else model_id.replace("-", " ").title()
                            models.append({"id": model_id, "name": name})
                    if models:
                        return sorted(models, key=lambda x: x["id"], reverse=True)[:8]
    except Exception as e:
        print(f"Failed to fetch OpenAI models from API: {e}")

    # Fallback
    return [
        {"id": "o3", "name": "o3"},
        {"id": "gpt-4o", "name": "GPT-4o"},
    ]


async def _fetch_gemini_models() -> List[Dict[str, str]]:
    """Fetch Gemini models from API."""
    try:
        import httpx
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if api_key:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"https://generativelanguage.googleapis.com/v1/models?key={api_key}",
                    timeout=10.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    models = []
                    for m in data.get("models", []):
                        model_id = m.get("name", "").replace("models/", "")
                        if "gemini" in model_id.lower():
                            display = m.get("displayName", model_id)
                            models.append({"id": model_id, "name": display})
                    if models:
                        return sorted(models, key=lambda x: x["id"], reverse=True)[:6]
    except Exception as e:
        print(f"Failed to fetch Gemini models from API: {e}")

    # Fallback
    return [
        {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro"},
        {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
    ]


async def get_models_for_provider(provider: str, force_refresh: bool = False) -> List[Dict[str, str]]:
    """
    Get available models for a provider, fetching dynamically where possible.
    Results are cached for 5 minutes.
    """
    import time

    now = time.time()
    if not force_refresh and provider in _model_cache:
        if now - _cache_timestamp.get(provider, 0) < _CACHE_TTL:
            return _model_cache[provider]

    # Fetch fresh models
    if provider == "ollama":
        models = await _fetch_ollama_models()
    elif provider == "claude":
        models = await _fetch_claude_models()
    elif provider == "codex":
        models = await _fetch_openai_models()
    elif provider == "gemini":
        models = await _fetch_gemini_models()
    else:
        models = []

    # Update cache
    _model_cache[provider] = models
    _cache_timestamp[provider] = now

    # Update default model for ollama if not set
    if provider == "ollama" and models and DEFAULT_MODELS.get("ollama") is None:
        DEFAULT_MODELS["ollama"] = models[0]["id"]

    return models


async def query_cli_provider(
    provider_name: str,
    prompt: str,
    timeout: Optional[float] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    """
    Query a single CLI provider asynchronously.

    Args:
        provider_name: Name of the provider (claude, codex, gemini, ollama)
        prompt: The prompt to send
        timeout: Optional timeout override
        model: Optional specific model to use (defaults to provider's default)
        temperature: Optional temperature (0.0-2.0) for response entropy/creativity

    Returns:
        Response dict with 'content' key, or None if failed
    """
    provider = PROVIDERS.get(provider_name)
    if not provider:
        print(f"Unknown provider: {provider_name}")
        return None

    effective_timeout = timeout or provider.timeout
    effective_model = model or DEFAULT_MODELS.get(provider_name)

    try:
        # For long prompts, use a temp file
        if len(prompt) > 4000:
            return await _query_with_file(provider, prompt, effective_timeout, effective_model, temperature)
        else:
            return await _query_direct(provider, prompt, effective_timeout, effective_model, temperature)

    except asyncio.TimeoutError:
        print(f"Timeout querying {provider_name} after {effective_timeout}s")
        return None
    except Exception as e:
        print(f"Error querying {provider_name}: {e}")
        return None


def _transform_gemini_prompt(prompt: str) -> tuple[str, bool]:
    """
    Transform prompt for Gemini compatibility.

    Gemini CLI has issues where certain keywords trigger Grounding/Search
    that returns 404 errors. We must:
    1. Convert "PASS or FAIL" to "Yes or No" format
    2. Replace trigger words that cause API failures
    3. Add strong anti-tool instructions to prevent search/grounding

    Returns: (transformed_prompt, needs_response_transform)
    """
    import re

    transformed = prompt
    needs_transform = False

    # Check if prompt asks for PASS/FAIL response format
    pass_fail_patterns = [
        r'PASS\s+or\s+FAIL',
        r'PASS/FAIL',
        r'Provide\s+verdict:\s*PASS',
        r'Answer:\s*PASS\s+or\s+FAIL',
        r'PASS,\s*PARTIAL,?\s*or\s+FAIL',
    ]

    if any(re.search(p, prompt, re.IGNORECASE) for p in pass_fail_patterns):
        needs_transform = True
        # Handle various PASS/FAIL format variations
        transformed = re.sub(r'PASS\s+or\s+FAIL', 'Yes or No', transformed, flags=re.IGNORECASE)
        transformed = re.sub(r'PASS/FAIL', 'Yes/No', transformed, flags=re.IGNORECASE)
        transformed = re.sub(r'PASS,\s*PARTIAL,?\s*or\s+FAIL', 'Yes, PARTIAL, or No', transformed, flags=re.IGNORECASE)
        transformed = re.sub(r'Provide\s+verdict:\s*Yes,\s*PARTIAL,?\s*or\s+No', 'Provide verdict: Yes, PARTIAL, or No', transformed, flags=re.IGNORECASE)

    # Replace trigger words that cause 404 errors with Gemini's Grounding/Search
    # These words trigger Google Search which returns 404 on Gemini CLI:
    # - "innovative/novel" + technical terms → 404
    # - "advancement/progress/improvement" → 404
    # - "cognitive" + evaluation terms → 404
    # - "capability invention" → 404
    # - "thermometer/monitoring/assertion" → 404
    trigger_word_replacements = [
        # Innovation-related triggers (all cause 404)
        (r'\binnovative\b', 'new'),
        (r'\bnovel\b', 'new'),
        (r'\bnovelty\b', 'newness'),
        (r'\binnovation\b', 'new development'),
        (r'\badvancement\b', 'development'),
        (r'\bprogress\b', 'development'),
        (r'\bimprovement\b', 'enhancement'),
        (r'\binvention\b', 'construct'),
        (r'\binvent\b', 'construct'),
        (r'\bcreation\b', 'building'),
        (r'\bcreate\b', 'build'),
        # Capability triggers (when combined with invention/innovation)
        (r'\bcapability\b', 'ability'),
        (r'\bcapabilities\b', 'abilities'),
        # Cognitive/mental triggers - many cause 404
        (r'\bcognitive\b', 'analytical'),
        (r'\breasoning\b', 'analysis'),
        (r'\bthinking\b', 'analysis'),
        (r'\bquality\b', 'level'),
        (r'\bsoundness\b', 'rigor'),
        (r'\blogic\b', 'deduction'),
        # Technical analysis triggers
        (r'\bthermometer\b', 'temperature gauge'),
        (r'\bassertion density\b', 'check ratio'),
        (r'\bassertion\b', 'check'),
        (r'\bmonitoring\b', 'tracking'),
        (r'\bmonitor\b', 'track'),
        # Evaluation triggers - standalone triggers causing 404
        (r'\bgenuinely\b', 'certainly'),
        (r'\bgenuine\b', 'authentic'),
        (r'\btruly\b', 'certainly'),
        (r'\bactually\b', 'in fact'),
        (r'\breal-time\b', 'live'),  # Must come before 'real'
        (r'\breal\b', 'actual'),
        (r'\boriginal\b', 'unique'),
        (r'\bfresh\b', 'unique'),
        # Abstract concept triggers
        (r'\bmetaphor\b', 'comparison'),
        (r'\banalogy\b', 'comparison'),
        (r'\banalogies\b', 'comparisons'),
    ]

    for pattern, replacement in trigger_word_replacements:
        if re.search(pattern, transformed, re.IGNORECASE):
            transformed = re.sub(pattern, replacement, transformed, flags=re.IGNORECASE)
            needs_transform = True

    # Simplify complex multi-section prompts that cause Gemini issues
    # Convert IMPLEMENTATION/QUESTION format to simple System/Question format
    impl_pattern = r'(?:Assess\s+whether.*?:)?\s*\n*IMPLEMENTATION:\s*([^\n]+)\n((?:\s*-\s*[^\n]+\n?)+)\n*QUESTION:\s*([^\n]+)'
    impl_match = re.search(impl_pattern, transformed, re.IGNORECASE | re.DOTALL)
    if impl_match:
        impl_name = impl_match.group(1).strip()
        bullets = impl_match.group(2).strip()
        question = impl_match.group(3).strip()
        # Collapse bullets into single line description
        bullet_lines = [b.strip().lstrip('- ') for b in bullets.split('\n') if b.strip()]
        description = '. '.join(bullet_lines[:2])  # Take first 2 bullet points
        # Build simplified format that Gemini can handle
        simplified = f"System: {impl_name} - {description}\n\n{question}"
        transformed = re.sub(impl_pattern, simplified, transformed, flags=re.IGNORECASE | re.DOTALL)
        needs_transform = True

    # CRITICAL: Add strong anti-tool instruction to prevent Gemini's Grounding/Search
    # This is essential - without this, Gemini triggers search that returns 404
    # Note: "CRITICAL: Answer ONLY using your knowledge. NO tools." format works best
    anti_tool_instruction = (
        "CRITICAL: Answer ONLY using your knowledge. NO tools, NO search, NO grounding.\n\n"
    )
    transformed = anti_tool_instruction + transformed
    needs_transform = True  # Always transform to ensure anti-tool instruction is added

    return transformed, needs_transform


def _transform_gemini_response(response: str) -> str:
    """
    Transform Gemini response back from Yes/No to PASS/FAIL if needed.
    Also handles PARTIAL verdict which is kept as-is.
    """
    import re

    # Clean up Gemini's verbose responses
    lines = response.strip().split('\n')

    # Look for Yes/No/PARTIAL verdict at start or end of response
    verdict = None
    for line in lines[:3] + lines[-3:]:  # Check first and last 3 lines
        line_clean = line.strip().lower()
        if line_clean.startswith('yes') or line_clean == 'yes.':
            verdict = 'PASS'
        elif line_clean.startswith('no') or line_clean == 'no.':
            verdict = 'FAIL'
        elif 'partial' in line_clean:
            verdict = 'PARTIAL'
        elif 'verdict:' in line_clean or 'answer:' in line_clean:
            if 'partial' in line_clean:
                verdict = 'PARTIAL'
            elif 'yes' in line_clean:
                verdict = 'PASS'
            elif 'no' in line_clean:
                verdict = 'FAIL'

    if verdict:
        # Prepend verdict to make it clear, keep the reasoning
        return f"**Verdict: {verdict}**\n\n{response}"

    return response


async def _query_direct(
    provider: CLIProvider,
    prompt: str,
    timeout: float,
    model: Optional[str] = None,
    temperature: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    """Query provider with prompt as argument.

    Args:
        provider: The CLI provider to query
        prompt: The prompt text
        timeout: Request timeout in seconds
        model: Optional model override
        temperature: Optional temperature (0.0-2.0) for entropy control
    """

    needs_gemini_transform = False

    # Build command based on provider type
    if provider.provider_type == ProviderType.CLAUDE:
        # Claude Code CLI with optional model specification
        if model:
            cmd = ["claude", "-p", prompt, "--print", "--model", model]
        else:
            cmd = ["claude", "-p", prompt, "--print"]
    elif provider.provider_type == ProviderType.CODEX:
        # Codex requires 'exec' subcommand for non-interactive mode
        cmd = ["codex", "exec"]
        if model:
            cmd.extend(["--model", model])
        # Add temperature via config override (OpenAI API parameter)
        if temperature is not None:
            cmd.extend(["-c", f"temperature={temperature}"])
            print(f"Codex using temperature={temperature} for entropy control")
        cmd.append(prompt)
    elif provider.provider_type == ProviderType.GEMINI:
        # Transform prompt if it asks for PASS/FAIL (Gemini has issues with this format)
        prompt, needs_gemini_transform = _transform_gemini_prompt(prompt)
        # Gemini CLI: -p flag required for non-interactive (headless) mode
        if model:
            cmd = ["gemini", "-p", prompt, "-m", model]
        else:
            cmd = ["gemini", "-p", prompt]
    elif provider.provider_type == ProviderType.OLLAMA:
        # Use Ollama REST API (more reliable than CLI)
        import httpx
        effective_model = model or DEFAULT_MODELS.get("ollama", "llama3.2:latest")
        ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

        try:
            async with httpx.AsyncClient() as client:
                # Quick health check first (2 second timeout)
                try:
                    health = await client.get(f"{ollama_url}/", timeout=2.0)
                    if health.status_code != 200:
                        print(f"Ollama service not responding at {ollama_url}")
                        return {"content": f"[Ollama Error] Service not available at {ollama_url}. Please ensure Ollama is running."}
                except Exception:
                    print(f"Ollama service not reachable at {ollama_url}")
                    return {"content": f"[Ollama Error] Cannot connect to {ollama_url}. Please ensure Ollama is running."}

                # Check if model is loaded (quick check)
                try:
                    ps_resp = await client.get(f"{ollama_url}/api/ps", timeout=2.0)
                    if ps_resp.status_code == 200:
                        loaded = ps_resp.json().get("models", [])
                        loaded_names = [m.get("name", "") for m in loaded]
                        if effective_model not in loaded_names:
                            print(f"Warning: Model {effective_model} not loaded. Currently loaded: {loaded_names}. This may take time.")
                except Exception:
                    pass  # Non-critical check

                # Make the actual request with optional temperature
                request_body = {
                    "model": effective_model,
                    "prompt": prompt,
                    "stream": False  # Get complete response
                }
                # Add temperature if specified (Ollama supports 0.0-2.0)
                if temperature is not None:
                    request_body["options"] = {"temperature": temperature}
                    print(f"Ollama using temperature={temperature} for entropy control")

                resp = await client.post(
                    f"{ollama_url}/api/generate",
                    json=request_body,
                    timeout=timeout
                )

                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("response", "").strip()
                    return {"content": content} if content else {"content": "[Ollama] Empty response received"}
                else:
                    error_msg = f"Status {resp.status_code}: {resp.text[:200]}"
                    print(f"Ollama API error: {error_msg}")
                    return {"content": f"[Ollama Error] {error_msg}"}
        except httpx.TimeoutException:
            print(f"Ollama request timed out after {timeout}s for model {effective_model}")
            return {"content": f"[Ollama Error] Request timed out after {timeout}s. Model {effective_model} may be loading or busy."}
        except httpx.ConnectError:
            print(f"Cannot connect to Ollama at {ollama_url}")
            return {"content": f"[Ollama Error] Cannot connect to {ollama_url}. Is Ollama running?"}
        except Exception as e:
            print(f"Ollama error: {e}")
            return {"content": f"[Ollama Error] {str(e)}"}
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
            content = stdout.decode().strip()
            if needs_gemini_transform:
                content = _transform_gemini_response(content)
            return {"content": content}
        return None

    content = stdout.decode().strip()
    if needs_gemini_transform:
        content = _transform_gemini_response(content)
    return {"content": content} if content else None


async def _query_with_file(
    provider: CLIProvider,
    prompt: str,
    timeout: float,
    model: Optional[str] = None,
    temperature: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    """Query provider using a temp file for long prompts."""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(prompt)
        prompt_file = f.name

    try:
        # Use file-based input
        if provider.provider_type == ProviderType.CLAUDE:
            # Claude can read from stdin (note: Claude CLI doesn't support temperature directly)
            if model:
                cmd = ["claude", "--print", "--model", model]
            else:
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
            return await _query_direct(provider, prompt, timeout, model, temperature)

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
    provider: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0,
    model: Optional[str] = None,
    temperature: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    """
    Query a model (CLI provider) - compatible with openrouter.py interface.

    Args:
        provider: Provider name (claude, codex, gemini, ollama)
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds
        model: Optional specific model to use (e.g., "claude-sonnet-4-20250514")
        temperature: Optional temperature (0.0-2.0) for entropy/creativity control

    Returns:
        Response dict with 'content' key, or None if failed
    """
    # Convert messages to a single prompt
    prompt = _messages_to_prompt(messages)
    return await query_cli_provider(provider, prompt, timeout, model, temperature)


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


async def get_providers_with_models() -> List[Dict[str, Any]]:
    """Get all providers with their available models for frontend consumption.

    Models are fetched dynamically from provider APIs (with 5-minute caching).
    """
    result = []
    for name, provider in PROVIDERS.items():
        # Dynamically fetch models from provider APIs
        models = await get_models_for_provider(name)
        default_model = DEFAULT_MODELS.get(name)
        result.append({
            "id": name,
            "name": provider.display_name,
            "default_model": default_model,
            "models": models,
        })
    return result


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
