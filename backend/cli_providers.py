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
    timeout: float
) -> Optional[Dict[str, Any]]:
    """Query provider with prompt as argument."""

    needs_gemini_transform = False

    # Build command based on provider type
    if provider.provider_type == ProviderType.CLAUDE:
        cmd = ["claude", "-p", prompt, "--print"]
    elif provider.provider_type == ProviderType.CODEX:
        # Codex requires 'exec' subcommand for non-interactive mode
        cmd = ["codex", "exec", prompt]
    elif provider.provider_type == ProviderType.GEMINI:
        # Transform prompt if it asks for PASS/FAIL (Gemini has issues with this format)
        prompt, needs_gemini_transform = _transform_gemini_prompt(prompt)
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
